import logging
from saml2.cache import Cache
def stale_sources_for_person(self, name_id, sources=None):
    """

        :param name_id: Identifier of the subject, a NameID instance
        :param sources: Sources for information about the subject
        :return:
        """
    if not sources:
        sources = self.cache.entities(name_id)
    sources = [m for m in sources if not self.cache.active(name_id, m)]
    return sources