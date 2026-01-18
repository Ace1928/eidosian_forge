from collections import defaultdict
import click
def cb_multilayer(ctx, param, value):
    """
    Transform layer options from strings ("1:a,1:b", "2:a,2:c,2:z") to
    {
    '1': ['a', 'b'],
    '2': ['a', 'c', 'z']
    }
    """
    out = defaultdict(list)
    for raw in value:
        for v in raw.split(','):
            ds, name = v.split(':')
            out[ds].append(name)
    return out