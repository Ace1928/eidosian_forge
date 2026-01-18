import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def gen_unpack_spec(self, tpl):
    """Generate a specification for '__rl_unpack_sequence__'.

		This spec is used to protect sequence unpacking.
		The primary goal of this spec is to tell which elements in a sequence
		are sequences again. These 'child' sequences have to be protected
		again.

		For example there is a sequence like this:
			(a, (b, c), (d, (e, f))) = g

		On a higher level the spec says:
			- There is a sequence of len 3
			- The element at index 1 is a sequence again with len 2
			- The element at index 2 is a sequence again with len 2
			  - The element at index 1 in this subsequence is a sequence again
				with len 2

		With this spec '__rl_unpack_sequence__' does something like this for
		protection (len checks are omitted):

			t = list(__rl_getiter__(g))
			t[1] = list(__rl_getiter__(t[1]))
			t[2] = list(__rl_getiter__(t[2]))
			t[2][1] = list(__rl_getiter__(t[2][1]))
			return t

		The 'real' spec for the case above is then:
			spec = {
				'min_len': 3,
				'childs': (
					(1, {'min_len': 2, 'childs': ()}),
					(2, {
							'min_len': 2,
							'childs': (
								(1, {'min_len': 2, 'childs': ()})
							)
						}
					)
				)
			}

		So finally the assignment above is converted into:
			(a, (b, c), (d, (e, f))) = __rl_unpack_sequence__(g, spec)
		"""
    spec = ast.Dict(keys=[], values=[])
    spec.keys.append(ast.Str('childs'))
    spec.values.append(ast.Tuple([], ast.Load()))
    min_len = len([ob for ob in tpl.elts if not self.is_starred(ob)])
    offset = 0
    for idx, val in enumerate(tpl.elts):
        if self.is_starred(val):
            offset = min_len + 1
        elif isinstance(val, ast.Tuple):
            el = ast.Tuple([], ast.Load())
            el.elts.append(ast.Num(idx - offset))
            el.elts.append(self.gen_unpack_spec(val))
            spec.values[0].elts.append(el)
    spec.keys.append(ast.Str('min_len'))
    spec.values.append(ast.Num(min_len))
    return spec