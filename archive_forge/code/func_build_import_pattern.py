from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, String, FromImport, Newline, Comma
from libfuturize.fixer_util import touch_import_top
from_import = u"from_import=import_from< 'from' %s 'import' (import_as_name< using=any 'as' renamed=any> | in_list=import_as_names< using=any* > | using='*' | using=NAME) >"
from_import_rename = u"from_import_rename=import_from< 'from' %s 'import' (%s | import_as_name< %s 'as' renamed=any > | in_list=import_as_names< any* (%s | import_as_name< %s 'as' renamed=any >) any* >) >"
def build_import_pattern(mapping1, mapping2):
    u"""
    mapping1: A dict mapping py3k modules to all possible py2k replacements
    mapping2: A dict mapping py2k modules to the things they do
    This builds a HUGE pattern to match all ways that things can be imported
    """
    yield (from_import % all_modules_subpattern())
    for py3k, py2k in mapping1.items():
        name, attr = py3k.split(u'.')
        s_name = simple_name % name
        s_attr = simple_attr % attr
        d_name = dotted_name % (s_name, s_attr)
        yield (name_import % d_name)
        yield (power_twoname % (s_name, s_attr))
        if attr == u'__init__':
            yield (name_import % s_name)
            yield (power_onename % s_name)
        yield (name_import_rename % d_name)
        yield (from_import_rename % (s_name, s_attr, s_attr, s_attr, s_attr))