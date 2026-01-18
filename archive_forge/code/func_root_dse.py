import dataclasses
import socket
import ssl
import threading
import typing as t
@property
def root_dse(self) -> RootDSE:
    if not self._root_dse:
        default_naming_context = ''
        subschema_subentry = ''
        supported_controls: t.List[str] = []
        for res in self._search_request(base_object='', scope=sansldap.SearchScope.BASE, filter=sansldap.FilterPresent('objectClass'), attributes=['defaultNamingContext', 'subschemaSubentry', 'supportedControl']):
            if not isinstance(res, sansldap.SearchResultEntry):
                continue
            for attr in res.attributes:
                if attr.name == 'defaultNamingContext':
                    default_naming_context = attr.values[0].decode('utf-8')
                elif attr.name == 'subschemaSubentry':
                    subschema_subentry = attr.values[0].decode('utf-8')
                elif attr.name == 'supportedControl':
                    supported_controls = [v.decode('utf-8') for v in attr.values]
        self._root_dse = RootDSE(default_naming_context=default_naming_context, subschema_subentry=subschema_subentry, supported_controls=supported_controls)
    return self._root_dse