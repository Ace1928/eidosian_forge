from parso.python import tree
from jedi import debug
from jedi.inference.helpers import is_string
def check_hasattr(node, suite):
    try:
        assert suite.start_pos <= jedi_name.start_pos < suite.end_pos
        assert node.type in ('power', 'atom_expr')
        base = node.children[0]
        assert base.type == 'name' and base.value == 'hasattr'
        trailer = node.children[1]
        assert trailer.type == 'trailer'
        arglist = trailer.children[1]
        assert arglist.type == 'arglist'
        from jedi.inference.arguments import TreeArguments
        args = TreeArguments(node_context.inference_state, node_context, arglist)
        unpacked_args = list(args.unpack())
        assert len(unpacked_args) == 2
        key, lazy_value = unpacked_args[1]
        names = list(lazy_value.infer())
        assert len(names) == 1 and is_string(names[0])
        assert names[0].get_safe_value() == payload[1].value
        key, lazy_value = unpacked_args[0]
        objects = lazy_value.infer()
        return payload[0] in objects
    except AssertionError:
        return False