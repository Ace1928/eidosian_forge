from numba.np.numpy_support import from_dtype
@cuda.jit(device=True)
def inner_warp_reduction(sm_partials, init):
    """
        Compute reduction within a single warp
        """
    tid = cuda.threadIdx.x
    warpid = tid // _WARPSIZE
    laneid = tid % _WARPSIZE
    sm_this = sm_partials[warpid, :]
    sm_this[laneid] = init
    cuda.syncwarp()
    width = _WARPSIZE // 2
    while width:
        if laneid < width:
            old = sm_this[laneid]
            sm_this[laneid] = reduce_op(old, sm_this[laneid + width])
        cuda.syncwarp()
        width //= 2