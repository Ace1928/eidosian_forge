from collections import namedtuple
import macaroonbakery.checkers as checkers
def legacy_namespace():
    """ Standard namespace for pre-version3 macaroons.
    """
    ns = checkers.Namespace(None)
    ns.register(checkers.STD_NAMESPACE, '')
    return ns