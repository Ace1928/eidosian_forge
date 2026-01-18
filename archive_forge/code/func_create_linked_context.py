import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
@staticmethod
def create_linked_context():

    def f():
        pass

    def g():
        pass

    def g_():
        pass

    def g__():
        pass
    context1 = contexts.Context()
    context2 = contexts.Context()
    context1.register_function(f)
    context1.register_function(g)
    context2.register_function(g_)
    context1['key'] = 'context1'
    context1['key1'] = 'context1'
    context2['key'] = 'context2'
    context2['key2'] = 'context2'
    context3 = context2.create_child_context()
    context3.register_function(g__)
    context3['key'] = 'context3'
    context2['key3'] = 'context3'
    return contexts.LinkedContext(parent_context=context1, linked_context=context3)