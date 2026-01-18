def make_test_application(global_conf, text=False, lint=False):
    from paste.deploy.converters import asbool
    text = asbool(text)
    lint = asbool(lint)
    app = TestApplication(global_conf=global_conf, text=text)
    if lint:
        from paste.lint import middleware
        app = middleware(app)
    return app