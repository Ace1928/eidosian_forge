from collections import Counter
from textwrap import dedent
from kombu.utils.encoding import bytes_to_str, safe_str
def repr_node(self, obj, level=1, fmt='{0}({1})'):
    output = [fmt.format(obj, self.valency_of(obj))]
    if obj in self:
        for other in self[obj]:
            d = fmt.format(other, self.valency_of(other))
            output.append('     ' * level + d)
            output.extend(self.repr_node(other, level + 1).split('\n')[1:])
    return '\n'.join(output)