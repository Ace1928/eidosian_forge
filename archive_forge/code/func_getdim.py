import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def getdim(self, xyz):
    i64 = ir.IntType(64)
    tid = self.builder.sext(self.tid(xyz), i64)
    ntid = self.builder.sext(self.ntid(xyz), i64)
    nctaid = self.builder.sext(self.ctaid(xyz), i64)
    res = self.builder.add(self.builder.mul(ntid, nctaid), tid)
    return res