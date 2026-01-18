import logging
from saml2.attribute_resolver import AttributeResolver
from saml2.saml import NAMEID_FORMAT_PERSISTENT
def do_aggregation(self, name_id):
    logger.info('** Do VO aggregation **\nSubjectID: %s, VO:%s', name_id, self._name)
    to_ask = self.members_to_ask(name_id)
    if to_ask:
        com_identifier = self.get_common_identifier(name_id)
        resolver = AttributeResolver(self.sp)
        for session_info in resolver.extend(com_identifier, self.sp.config.entityid, to_ask):
            _ = self._cache_session(session_info)
        logger.info('>Issuers: %s', self.sp.users.issuers_of_info(name_id))
        logger.info('AVA: %s', self.sp.users.get_identity(name_id))
        return True
    else:
        return False