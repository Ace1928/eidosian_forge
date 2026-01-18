import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def test_args_options(self):
    for o in ('max', 'native'):
        if o == 'native' and self.cc_name() == 'msvc':
            continue
        self.expect(o, trap_files='.*cpu_(sse|vsx|neon|vx).c', x86='', ppc64='', armhf='', s390x='')
        self.expect(o, trap_files='.*cpu_(sse3|vsx2|neon_vfpv4|vxe).c', x86='sse sse2', ppc64='vsx', armhf='neon neon_fp16', aarch64='', ppc64le='', s390x='vx')
        self.expect(o, trap_files='.*cpu_(popcnt|vsx3).c', x86='sse .* sse41', ppc64='vsx vsx2', armhf='neon neon_fp16 .* asimd .*', s390x='vx vxe vxe2')
        self.expect(o, x86_gcc='.* xop fma4 .* avx512f .* avx512_knl avx512_knm avx512_skx .*', x86_icc='.* avx512f .* avx512_knl avx512_knm avx512_skx .*', x86_iccw='.* avx512f .* avx512_knl avx512_knm avx512_skx .*', x86_msvc='.* xop fma4 .* avx512f .* avx512_skx .*', armhf='.* asimd asimdhp asimddp .*', ppc64='vsx vsx2 vsx3 vsx4.*', s390x='vx vxe vxe2.*')
    self.expect('min', x86='sse sse2', x64='sse sse2 sse3', armhf='', aarch64='neon neon_fp16 .* asimd', ppc64='', ppc64le='vsx vsx2', s390x='')
    self.expect('min', trap_files='.*cpu_(sse2|vsx2).c', x86='', ppc64le='')
    try:
        self.expect('native', trap_flags='.*(-march=native|-xHost|/QxHost|-mcpu=a64fx).*', x86='.*', ppc64='.*', armhf='.*', s390x='.*', aarch64='.*')
        if self.march() != 'unknown':
            raise AssertionError('excepted an exception for %s' % self.march())
    except DistutilsError:
        if self.march() == 'unknown':
            raise AssertionError('excepted no exceptions')