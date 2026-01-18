import triton
import triton.language as tl
from xformers.triton.k_activations import (
@triton.heuristics({'SIZE_RAND_BLOCK': lambda args: args['BLOCK_N'] * args['BLOCK_M']})
@triton.autotune(configs=_configs, key=['M', 'N', 'is_fp16'])
@triton.jit
def k_dropout_fw(Y, X, BIAS, SEEDS, stride, M, N, p: tl.constexpr, is_fp16: tl.constexpr, ACTIVATION: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SIZE_RAND_BLOCK: tl.constexpr, USE_BIAS: tl.constexpr):
    """
    Apply dropout on an input tensor
    Y : Output  (M, N)
    X : Input   (M, N)
    BIAS        (N,)
    SEEDS       (M,)
    p : dropout probability
    """
    row_id = tl.program_id(axis=0)
    rows = row_id * BLOCK_M + tl.arange(0, BLOCK_M)
    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = X + rows[:, None] * stride + cols[None, :]
    y_ptrs = Y + rows[:, None] * stride + cols[None, :]
    col_mask = cols[None, :] < N
    p_scale = 1.0 / (1.0 - p)
    if USE_BIAS:
        b_ptrs = BIAS + cols[None, :]
        bias = tl.load(b_ptrs, mask=cols[None, :] < N, other=0.0)
    else:
        bias = x_ptrs
    block_mask = (rows[:, None] < M) & col_mask
    x = tl.load(x_ptrs, mask=block_mask, other=0.0)
    if USE_BIAS:
        x += bias
    if ACTIVATION == 1:
        x = relu(x)
    elif ACTIVATION == 2:
        x = leaky_relu(x)
    elif ACTIVATION == 3:
        x = gelu(x)
    elif ACTIVATION == 4:
        x = squared_relu(x)
    elif ACTIVATION == 5:
        x = smelu(x)
    rand_offsets = tl.arange(0, SIZE_RAND_BLOCK)
    seed_int = tl.load(SEEDS + col_id)
    r = tl.rand(seed_int, rand_offsets)
    keep_mask = r > p
    keep = tl.view(keep_mask, x.shape)
    output = tl.where(keep, (x * p_scale).to(x.dtype), 0.0)
    tl.store(y_ptrs, output, mask=block_mask)