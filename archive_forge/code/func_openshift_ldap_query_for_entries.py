from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def openshift_ldap_query_for_entries(connection, qry, unique_entry=True):
    derefAlias = qry.pop('derefAlias', None)
    if derefAlias:
        ldap.set_option(ldap.OPT_DEREF, derefAlias)
    try:
        result = connection.search_ext_s(**qry)
        if not result or len(result) == 0:
            return (None, "Entry not found for base='{0}' and filter='{1}'".format(qry['base'], qry['filterstr']))
        if len(result) > 1 and unique_entry:
            if qry.get('scope') == ldap.SCOPE_BASE:
                return (None, 'multiple entries found matching dn={0}: {1}'.format(qry['base'], result))
            else:
                return (None, 'multiple entries found matching filter {0}: {1}'.format(qry['filterstr'], result))
        return (result, None)
    except ldap.NO_SUCH_OBJECT:
        return (None, "search for entry with base dn='{0}' refers to a non-existent entry".format(qry['base']))