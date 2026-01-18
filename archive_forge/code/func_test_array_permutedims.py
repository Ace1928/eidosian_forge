import itertools
import random
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import _af_invert
from sympy.testing.pytest import raises
from sympy.core.function import diff
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.tensor.array import Array, ImmutableDenseNDimArray, ImmutableSparseNDimArray, MutableSparseNDimArray
from sympy.tensor.array.arrayop import tensorproduct, tensorcontraction, derive_by_array, permutedims, Flatten, \
def test_array_permutedims():
    sa = symbols('a0:144')
    for ArrayType in [ImmutableDenseNDimArray, ImmutableSparseNDimArray]:
        m1 = ArrayType(sa[:6], (2, 3))
        assert permutedims(m1, (1, 0)) == transpose(m1)
        assert m1.tomatrix().T == permutedims(m1, (1, 0)).tomatrix()
        assert m1.tomatrix().T == transpose(m1).tomatrix()
        assert m1.tomatrix().C == conjugate(m1).tomatrix()
        assert m1.tomatrix().H == adjoint(m1).tomatrix()
        assert m1.tomatrix().T == m1.transpose().tomatrix()
        assert m1.tomatrix().C == m1.conjugate().tomatrix()
        assert m1.tomatrix().H == m1.adjoint().tomatrix()
        raises(ValueError, lambda: permutedims(m1, (0,)))
        raises(ValueError, lambda: permutedims(m1, (0, 0)))
        raises(ValueError, lambda: permutedims(m1, (1, 2, 0)))
        dims = 6
        shape = [random.randint(1, 5) for i in range(dims)]
        elems = [random.random() for i in range(tensorproduct(*shape))]
        ra = ArrayType(elems, shape)
        perm = list(range(dims))
        random.shuffle(perm)
        assert permutedims(permutedims(ra, perm), _af_invert(perm)) == ra
        assert permutedims(ra, perm).shape == tuple(Permutation(perm)(shape))
        z = ArrayType.zeros(4, 5, 6, 7)
        assert permutedims(z, (2, 3, 1, 0)).shape == (6, 7, 5, 4)
        assert permutedims(z, [2, 3, 1, 0]).shape == (6, 7, 5, 4)
        assert permutedims(z, Permutation([2, 3, 1, 0])).shape == (6, 7, 5, 4)
        po = ArrayType(sa, [2, 2, 3, 3, 2, 2])
        raises(ValueError, lambda: permutedims(po, (1, 1)))
        raises(ValueError, lambda: po.transpose())
        raises(ValueError, lambda: po.adjoint())
        assert permutedims(po, reversed(range(po.rank()))) == ArrayType([[[[[[sa[0], sa[72]], [sa[36], sa[108]]], [[sa[12], sa[84]], [sa[48], sa[120]]], [[sa[24], sa[96]], [sa[60], sa[132]]]], [[[sa[4], sa[76]], [sa[40], sa[112]]], [[sa[16], sa[88]], [sa[52], sa[124]]], [[sa[28], sa[100]], [sa[64], sa[136]]]], [[[sa[8], sa[80]], [sa[44], sa[116]]], [[sa[20], sa[92]], [sa[56], sa[128]]], [[sa[32], sa[104]], [sa[68], sa[140]]]]], [[[[sa[2], sa[74]], [sa[38], sa[110]]], [[sa[14], sa[86]], [sa[50], sa[122]]], [[sa[26], sa[98]], [sa[62], sa[134]]]], [[[sa[6], sa[78]], [sa[42], sa[114]]], [[sa[18], sa[90]], [sa[54], sa[126]]], [[sa[30], sa[102]], [sa[66], sa[138]]]], [[[sa[10], sa[82]], [sa[46], sa[118]]], [[sa[22], sa[94]], [sa[58], sa[130]]], [[sa[34], sa[106]], [sa[70], sa[142]]]]]], [[[[[sa[1], sa[73]], [sa[37], sa[109]]], [[sa[13], sa[85]], [sa[49], sa[121]]], [[sa[25], sa[97]], [sa[61], sa[133]]]], [[[sa[5], sa[77]], [sa[41], sa[113]]], [[sa[17], sa[89]], [sa[53], sa[125]]], [[sa[29], sa[101]], [sa[65], sa[137]]]], [[[sa[9], sa[81]], [sa[45], sa[117]]], [[sa[21], sa[93]], [sa[57], sa[129]]], [[sa[33], sa[105]], [sa[69], sa[141]]]]], [[[[sa[3], sa[75]], [sa[39], sa[111]]], [[sa[15], sa[87]], [sa[51], sa[123]]], [[sa[27], sa[99]], [sa[63], sa[135]]]], [[[sa[7], sa[79]], [sa[43], sa[115]]], [[sa[19], sa[91]], [sa[55], sa[127]]], [[sa[31], sa[103]], [sa[67], sa[139]]]], [[[sa[11], sa[83]], [sa[47], sa[119]]], [[sa[23], sa[95]], [sa[59], sa[131]]], [[sa[35], sa[107]], [sa[71], sa[143]]]]]]])
        assert permutedims(po, (1, 0, 2, 3, 4, 5)) == ArrayType([[[[[[sa[0], sa[1]], [sa[2], sa[3]]], [[sa[4], sa[5]], [sa[6], sa[7]]], [[sa[8], sa[9]], [sa[10], sa[11]]]], [[[sa[12], sa[13]], [sa[14], sa[15]]], [[sa[16], sa[17]], [sa[18], sa[19]]], [[sa[20], sa[21]], [sa[22], sa[23]]]], [[[sa[24], sa[25]], [sa[26], sa[27]]], [[sa[28], sa[29]], [sa[30], sa[31]]], [[sa[32], sa[33]], [sa[34], sa[35]]]]], [[[[sa[72], sa[73]], [sa[74], sa[75]]], [[sa[76], sa[77]], [sa[78], sa[79]]], [[sa[80], sa[81]], [sa[82], sa[83]]]], [[[sa[84], sa[85]], [sa[86], sa[87]]], [[sa[88], sa[89]], [sa[90], sa[91]]], [[sa[92], sa[93]], [sa[94], sa[95]]]], [[[sa[96], sa[97]], [sa[98], sa[99]]], [[sa[100], sa[101]], [sa[102], sa[103]]], [[sa[104], sa[105]], [sa[106], sa[107]]]]]], [[[[[sa[36], sa[37]], [sa[38], sa[39]]], [[sa[40], sa[41]], [sa[42], sa[43]]], [[sa[44], sa[45]], [sa[46], sa[47]]]], [[[sa[48], sa[49]], [sa[50], sa[51]]], [[sa[52], sa[53]], [sa[54], sa[55]]], [[sa[56], sa[57]], [sa[58], sa[59]]]], [[[sa[60], sa[61]], [sa[62], sa[63]]], [[sa[64], sa[65]], [sa[66], sa[67]]], [[sa[68], sa[69]], [sa[70], sa[71]]]]], [[[[sa[108], sa[109]], [sa[110], sa[111]]], [[sa[112], sa[113]], [sa[114], sa[115]]], [[sa[116], sa[117]], [sa[118], sa[119]]]], [[[sa[120], sa[121]], [sa[122], sa[123]]], [[sa[124], sa[125]], [sa[126], sa[127]]], [[sa[128], sa[129]], [sa[130], sa[131]]]], [[[sa[132], sa[133]], [sa[134], sa[135]]], [[sa[136], sa[137]], [sa[138], sa[139]]], [[sa[140], sa[141]], [sa[142], sa[143]]]]]]])
        assert permutedims(po, (0, 2, 1, 4, 3, 5)) == ArrayType([[[[[[sa[0], sa[1]], [sa[4], sa[5]], [sa[8], sa[9]]], [[sa[2], sa[3]], [sa[6], sa[7]], [sa[10], sa[11]]]], [[[sa[36], sa[37]], [sa[40], sa[41]], [sa[44], sa[45]]], [[sa[38], sa[39]], [sa[42], sa[43]], [sa[46], sa[47]]]]], [[[[sa[12], sa[13]], [sa[16], sa[17]], [sa[20], sa[21]]], [[sa[14], sa[15]], [sa[18], sa[19]], [sa[22], sa[23]]]], [[[sa[48], sa[49]], [sa[52], sa[53]], [sa[56], sa[57]]], [[sa[50], sa[51]], [sa[54], sa[55]], [sa[58], sa[59]]]]], [[[[sa[24], sa[25]], [sa[28], sa[29]], [sa[32], sa[33]]], [[sa[26], sa[27]], [sa[30], sa[31]], [sa[34], sa[35]]]], [[[sa[60], sa[61]], [sa[64], sa[65]], [sa[68], sa[69]]], [[sa[62], sa[63]], [sa[66], sa[67]], [sa[70], sa[71]]]]]], [[[[[sa[72], sa[73]], [sa[76], sa[77]], [sa[80], sa[81]]], [[sa[74], sa[75]], [sa[78], sa[79]], [sa[82], sa[83]]]], [[[sa[108], sa[109]], [sa[112], sa[113]], [sa[116], sa[117]]], [[sa[110], sa[111]], [sa[114], sa[115]], [sa[118], sa[119]]]]], [[[[sa[84], sa[85]], [sa[88], sa[89]], [sa[92], sa[93]]], [[sa[86], sa[87]], [sa[90], sa[91]], [sa[94], sa[95]]]], [[[sa[120], sa[121]], [sa[124], sa[125]], [sa[128], sa[129]]], [[sa[122], sa[123]], [sa[126], sa[127]], [sa[130], sa[131]]]]], [[[[sa[96], sa[97]], [sa[100], sa[101]], [sa[104], sa[105]]], [[sa[98], sa[99]], [sa[102], sa[103]], [sa[106], sa[107]]]], [[[sa[132], sa[133]], [sa[136], sa[137]], [sa[140], sa[141]]], [[sa[134], sa[135]], [sa[138], sa[139]], [sa[142], sa[143]]]]]]])
        po2 = po.reshape(4, 9, 2, 2)
        assert po2 == ArrayType([[[[sa[0], sa[1]], [sa[2], sa[3]]], [[sa[4], sa[5]], [sa[6], sa[7]]], [[sa[8], sa[9]], [sa[10], sa[11]]], [[sa[12], sa[13]], [sa[14], sa[15]]], [[sa[16], sa[17]], [sa[18], sa[19]]], [[sa[20], sa[21]], [sa[22], sa[23]]], [[sa[24], sa[25]], [sa[26], sa[27]]], [[sa[28], sa[29]], [sa[30], sa[31]]], [[sa[32], sa[33]], [sa[34], sa[35]]]], [[[sa[36], sa[37]], [sa[38], sa[39]]], [[sa[40], sa[41]], [sa[42], sa[43]]], [[sa[44], sa[45]], [sa[46], sa[47]]], [[sa[48], sa[49]], [sa[50], sa[51]]], [[sa[52], sa[53]], [sa[54], sa[55]]], [[sa[56], sa[57]], [sa[58], sa[59]]], [[sa[60], sa[61]], [sa[62], sa[63]]], [[sa[64], sa[65]], [sa[66], sa[67]]], [[sa[68], sa[69]], [sa[70], sa[71]]]], [[[sa[72], sa[73]], [sa[74], sa[75]]], [[sa[76], sa[77]], [sa[78], sa[79]]], [[sa[80], sa[81]], [sa[82], sa[83]]], [[sa[84], sa[85]], [sa[86], sa[87]]], [[sa[88], sa[89]], [sa[90], sa[91]]], [[sa[92], sa[93]], [sa[94], sa[95]]], [[sa[96], sa[97]], [sa[98], sa[99]]], [[sa[100], sa[101]], [sa[102], sa[103]]], [[sa[104], sa[105]], [sa[106], sa[107]]]], [[[sa[108], sa[109]], [sa[110], sa[111]]], [[sa[112], sa[113]], [sa[114], sa[115]]], [[sa[116], sa[117]], [sa[118], sa[119]]], [[sa[120], sa[121]], [sa[122], sa[123]]], [[sa[124], sa[125]], [sa[126], sa[127]]], [[sa[128], sa[129]], [sa[130], sa[131]]], [[sa[132], sa[133]], [sa[134], sa[135]]], [[sa[136], sa[137]], [sa[138], sa[139]]], [[sa[140], sa[141]], [sa[142], sa[143]]]]])
        assert permutedims(po2, (3, 2, 0, 1)) == ArrayType([[[[sa[0], sa[4], sa[8], sa[12], sa[16], sa[20], sa[24], sa[28], sa[32]], [sa[36], sa[40], sa[44], sa[48], sa[52], sa[56], sa[60], sa[64], sa[68]], [sa[72], sa[76], sa[80], sa[84], sa[88], sa[92], sa[96], sa[100], sa[104]], [sa[108], sa[112], sa[116], sa[120], sa[124], sa[128], sa[132], sa[136], sa[140]]], [[sa[2], sa[6], sa[10], sa[14], sa[18], sa[22], sa[26], sa[30], sa[34]], [sa[38], sa[42], sa[46], sa[50], sa[54], sa[58], sa[62], sa[66], sa[70]], [sa[74], sa[78], sa[82], sa[86], sa[90], sa[94], sa[98], sa[102], sa[106]], [sa[110], sa[114], sa[118], sa[122], sa[126], sa[130], sa[134], sa[138], sa[142]]]], [[[sa[1], sa[5], sa[9], sa[13], sa[17], sa[21], sa[25], sa[29], sa[33]], [sa[37], sa[41], sa[45], sa[49], sa[53], sa[57], sa[61], sa[65], sa[69]], [sa[73], sa[77], sa[81], sa[85], sa[89], sa[93], sa[97], sa[101], sa[105]], [sa[109], sa[113], sa[117], sa[121], sa[125], sa[129], sa[133], sa[137], sa[141]]], [[sa[3], sa[7], sa[11], sa[15], sa[19], sa[23], sa[27], sa[31], sa[35]], [sa[39], sa[43], sa[47], sa[51], sa[55], sa[59], sa[63], sa[67], sa[71]], [sa[75], sa[79], sa[83], sa[87], sa[91], sa[95], sa[99], sa[103], sa[107]], [sa[111], sa[115], sa[119], sa[123], sa[127], sa[131], sa[135], sa[139], sa[143]]]]])
    for SparseArrayType in [ImmutableSparseNDimArray, MutableSparseNDimArray]:
        A = SparseArrayType({1: 1, 10000: 2}, (10000, 20000, 10000))
        assert permutedims(A, (0, 1, 2)) == A
        assert permutedims(A, (1, 0, 2)) == SparseArrayType({1: 1, 100000000: 2}, (20000, 10000, 10000))
        B = SparseArrayType({1: 1, 20000: 2}, (10000, 20000))
        assert B.transpose() == SparseArrayType({10000: 1, 1: 2}, (20000, 10000))