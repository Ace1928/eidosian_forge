from IPython.core import inputtransformer2 as ipt2
def test_classic_prompt():
    for sample, expected in [CLASSIC_PROMPT, CLASSIC_PROMPT_L2]:
        assert ipt2.classic_prompt(sample.splitlines(keepends=True)) == expected.splitlines(keepends=True)