import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
def get_ha1_dict(user_ha1_dict):
    """Returns a get_ha1 function which obtains a HA1 password hash from a
    dictionary of the form: {username : HA1}.

    If you want a dictionary-based authentication scheme, but with
    pre-computed HA1 hashes instead of plain-text passwords, use
    get_ha1_dict(my_userha1_dict) as the value for the get_ha1
    argument to digest_auth().
    """

    def get_ha1(realm, username):
        return user_ha1_dict.get(username)
    return get_ha1