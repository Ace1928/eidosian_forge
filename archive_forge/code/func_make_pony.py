from paste.request import construct_url
def make_pony(app, global_conf):
    """
    Adds pony power to any application, at /pony
    """
    return PonyMiddleware(app)