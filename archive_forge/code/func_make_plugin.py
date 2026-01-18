from urllib.parse import urlencode
from paste.httpexceptions import HTTPFound
from paste.httpheaders import CONTENT_LENGTH
from paste.httpheaders import CONTENT_TYPE
from paste.httpheaders import LOCATION
from paste.request import construct_url
from paste.request import parse_dict_querystring
from paste.request import parse_formvars
from repoze.who.interfaces import IChallenger
from repoze.who.interfaces import IIdentifier
from repoze.who.plugins.form import FormPlugin
from zope.interface import implements
def make_plugin(login_form_qs='__do_login', rememberer_name=None, form=None):
    if rememberer_name is None:
        raise ValueError('must include rememberer key (name of another IIdentifier plugin)')
    if form is not None:
        with open(form) as f:
            form = f.read()
    plugin = FormHiddenPlugin(login_form_qs, rememberer_name, form)
    return plugin