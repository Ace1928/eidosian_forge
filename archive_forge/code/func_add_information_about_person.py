import logging
from saml2.cache import Cache
def add_information_about_person(self, session_info):
    """If there already are information from this source in the cache
        this function will overwrite that information"""
    session_info = dict(session_info)
    name_id = session_info['name_id']
    issuer = session_info.pop('issuer')
    self.cache.set(name_id, issuer, session_info, session_info['not_on_or_after'])
    return name_id