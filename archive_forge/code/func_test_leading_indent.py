from IPython.core import inputtransformer2 as ipt2
def test_leading_indent():
    for sample, expected in [INDENT_SPACES, INDENT_TABS]:
        assert ipt2.leading_indent(sample.splitlines(keepends=True)) == expected.splitlines(keepends=True)