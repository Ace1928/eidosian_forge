from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template
def describe_template_sets():
    """
    Print the available template sets in this demo, with a short description"
    """
    import inspect
    import sys
    templatesets = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    for name, obj in templatesets:
        if name == 'describe_template_sets':
            continue
        print(name, obj.__doc__, '\n')