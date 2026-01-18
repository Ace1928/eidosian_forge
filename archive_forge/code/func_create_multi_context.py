import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
@staticmethod
def create_multi_context():

    def f():
        pass

    def f_():
        pass

    def f__():
        pass

    def f___():
        pass
    context = contexts.Context()
    context2 = context.create_child_context()
    context3 = context2.create_child_context()
    context4 = contexts.Context()
    context5 = context4.create_child_context()
    mc = contexts.MultiContext([context3, context5])
    context.register_function(f)
    context2.register_function(f___)
    context4.register_function(f_)
    context5.register_function(f__)
    context3['key'] = 'context3'
    context5['key'] = 'context5'
    context4['key2'] = 'context4'
    context['key3'] = 'context1'
    mc['key4'] = 'mc'
    return mc