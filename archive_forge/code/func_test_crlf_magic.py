from IPython.core import inputtransformer2 as ipt2
def test_crlf_magic():
    for sample, expected in [CRLF_MAGIC]:
        assert ipt2.cell_magic(sample) == expected