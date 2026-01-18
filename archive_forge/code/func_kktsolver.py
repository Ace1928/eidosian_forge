import sys
def kktsolver(x, z, W):
    f, Df, H = F(x, z)
    return factor(W, H, Df[1:, :])