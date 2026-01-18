from oslo_config import cfg
import paste.urlmap
def pipeline_factory(loader, global_conf, **local_conf):
    """A paste pipeline replica that keys off of deployment flavor."""
    pipeline = local_conf[CONF.paste_deploy.flavor or 'default']
    pipeline = pipeline.split()
    filters = [loader.get_filter(n) for n in pipeline[:-1]]
    app = loader.get_app(pipeline[-1])
    filters.reverse()
    for filter in filters:
        app = filter(app)
    return app