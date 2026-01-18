from textwrap import dedent
from pandas.util._decorators import doc
def test_doc_template_from_func():
    docstr = dedent('\n        This is the cummax method.\n\n        It computes the cumulative maximum.\n        ')
    assert cummax.__doc__ == docstr