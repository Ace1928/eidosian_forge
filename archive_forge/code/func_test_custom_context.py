import zmq
def test_custom_context():
    ctx = CustomContext('s')
    assert isinstance(ctx, CustomContext)
    assert ctx.extra_arg == 's'
    s = ctx.socket(zmq.PUSH, custom_attr=10)
    assert isinstance(s, CustomSocket)
    assert s.custom_attr == 10
    assert s.context is ctx
    assert s.type == zmq.PUSH
    s.close()
    ctx.term()