import triton
import triton.language as tl
@triton.jit
def layer_norm_bwd_dx_fused(DX, DY, DW, DB, X, W, M, V, Lock, stride, N, affine: tl.constexpr, GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptrs = X + row * stride + cols
    dy_ptrs = DY + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0)
    dy = tl.load(dy_ptrs, mask=mask, other=0)
    mean = tl.load(M + row)
    rstd = tl.load(V + row)
    xhat = (x - mean) * rstd
    if affine:
        w = tl.load(W + cols, mask=mask, other=0)
        wdy = w * dy
    else:
        wdy = dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    mean2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * mean1 + mean2)) * rstd
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    dx_ptrs = DX + row * stride + cols
    tl.store(dx_ptrs, dx, mask=mask)
    if affine:
        partial_dw = (dy * xhat).to(w.dtype)
        partial_db = dy.to(w.dtype)
        lock_id = row % GROUP_SIZE_M
        Lock += lock_id
        Count = Lock + GROUP_SIZE_M
        while tl.atomic_cas(Lock, 0, 1) == 1:
            pass
        count = tl.load(Count)
        dw_ptrs = DW + lock_id * N + cols
        db_ptrs = DB + lock_id * N + cols
        if count == 0:
            tl.atomic_xchg(Count, 1)
        else:
            partial_dw += tl.load(dw_ptrs, mask=mask, other=0.0)
            partial_db += tl.load(db_ptrs, mask=mask, other=0.0)
        tl.store(dw_ptrs, partial_dw, mask=mask)
        tl.store(db_ptrs, partial_db, mask=mask)
        tl.atomic_xchg(Lock, 0)