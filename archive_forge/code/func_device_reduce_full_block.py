from numba.np.numpy_support import from_dtype
@cuda.jit(device=True)
def device_reduce_full_block(arr, partials, sm_partials):
    """
        Partially reduce `arr` into `partials` using `sm_partials` as working
        space.  The algorithm goes like:

            array chunks of 128:  |   0 | 128 | 256 | 384 | 512 |
                        block-0:  |   x |     |     |   x |     |
                        block-1:  |     |   x |     |     |   x |
                        block-2:  |     |     |   x |     |     |

        The array is divided into chunks of 128 (size of a threadblock).
        The threadblocks consumes the chunks in roundrobin scheduling.
        First, a threadblock loads a chunk into temp memory.  Then, all
        subsequent chunks are combined into the temp memory.

        Once all chunks are processed.  Inner-block reduction is performed
        on the temp memory.  So that, there will just be one scalar result
        per block.  The result from each block is stored to `partials` at
        the dedicated slot.
        """
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blksz = cuda.blockDim.x
    gridsz = cuda.gridDim.x
    start = tid + blksz * blkid
    stop = arr.size
    step = blksz * gridsz
    tmp = arr[start]
    for i in range(start + step, stop, step):
        tmp = reduce_op(tmp, arr[i])
    cuda.syncthreads()
    inner_warp_reduction(sm_partials, tmp)
    cuda.syncthreads()
    if tid < 2:
        sm_partials[tid, 0] = reduce_op(sm_partials[tid, 0], sm_partials[tid + 2, 0])
        cuda.syncwarp()
    if tid == 0:
        partials[blkid] = reduce_op(sm_partials[0, 0], sm_partials[1, 0])