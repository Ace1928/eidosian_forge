from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import test_components as tc
from fire import testutils
from fire import trace
import mock
import six
def testCustomSerialize(self):

    def serialize(x):
        if isinstance(x, list):
            return ', '.join((str(xi) for xi in x))
        if isinstance(x, dict):
            return ', '.join(('{}={!r}'.format(k, v) for k, v in sorted(x.items())))
        if x == 'special':
            return ['SURPRISE!!', "I'm a list!"]
        return x
    ident = lambda x: x
    with self.assertOutputMatches(stdout='a, b', stderr=None):
        _ = core.Fire(ident, command=['[a,b]'], serialize=serialize)
    with self.assertOutputMatches(stdout='a=5, b=6', stderr=None):
        _ = core.Fire(ident, command=['{a:5,b:6}'], serialize=serialize)
    with self.assertOutputMatches(stdout='asdf', stderr=None):
        _ = core.Fire(ident, command=['asdf'], serialize=serialize)
    with self.assertOutputMatches(stdout="SURPRISE!!\nI'm a list!\n", stderr=None):
        _ = core.Fire(ident, command=['special'], serialize=serialize)
    with self.assertRaises(core.FireError):
        core.Fire(ident, command=['asdf'], serialize=55)