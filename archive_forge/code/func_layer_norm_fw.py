import triton
import triton.language as tl
@triton.jit
def layer_norm_fw(X, Y, W, B, M, V, stride, N, eps, affine: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    Fused layernorm kernel over a 3d tensor.
    The layer norm is applied over the last dimension.

    Compute
        y = (x - E(x))/(sqrt(var(x) + epsilon)) * gamma + beta
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_zm = tl.where(mask, x - mean, 0.0)
    tl.store(M + row, mean)
    x_var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(x_var + eps)
    y = x_zm * rstd
    tl.store(V + row, rstd)
    mask = cols < N
    if affine:
        w = tl.load(W + cols, mask=mask, other=1.0)
        b = tl.load(B + cols, mask=mask, other=0.0)
        y = y * w + b
    y_ptrs = Y + row * stride + cols
    tl.store(y_ptrs, y, mask=mask)