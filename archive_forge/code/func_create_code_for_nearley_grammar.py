import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
def create_code_for_nearley_grammar(g, start, builtin_path, folder_path, es6=False):
    import js2py
    emit_code = []

    def emit(x=None):
        if x:
            emit_code.append(x)
        emit_code.append('\n')
    js_code = ['function id(x) {return x[0];}']
    n2l = NearleyToLark()
    rule_defs = _nearley_to_lark(g, builtin_path, n2l, js_code, folder_path, set())
    lark_g = '\n'.join(rule_defs)
    lark_g += '\n' + '\n'.join(('!%s: %s' % item for item in n2l.extra_rules.items()))
    emit('from lark import Lark, Transformer')
    emit()
    emit('grammar = ' + repr(lark_g))
    emit()
    for alias, code in n2l.alias_js_code.items():
        js_code.append('%s = (%s);' % (alias, code))
    if es6:
        emit(js2py.translate_js6('\n'.join(js_code)))
    else:
        emit(js2py.translate_js('\n'.join(js_code)))
    emit('class TransformNearley(Transformer):')
    for alias in n2l.alias_js_code:
        emit("    %s = var.get('%s').to_python()" % (alias, alias))
    emit('    __default__ = lambda self, n, c, m: c if c else None')
    emit()
    emit('parser = Lark(grammar, start="n_%s", maybe_placeholders=False)' % start)
    emit('def parse(text):')
    emit('    return TransformNearley().transform(parser.parse(text))')
    return ''.join(emit_code)