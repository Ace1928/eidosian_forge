import os
import string
def soap_response(**kwargs):
    kwargs.setdefault('provider', 'https://idp.testshib.org/idp/shibboleth')
    kwargs.setdefault('consumer', 'https://openstack4.local/Shibboleth.sso/SAML2/ECP')
    kwargs.setdefault('issuer', 'https://openstack4.local/shibboleth')
    return template('soap_response.xml', **kwargs).encode('utf-8')