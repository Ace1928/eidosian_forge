from textwrap import dedent
from pandas.util._decorators import doc
def test_docstring_formatting():
    docstr = dedent('\n        This is the cumsum method.\n\n        It computes the cumulative sum.\n        ')
    assert cumsum.__doc__ == docstr