import keystone.conf
def symptom_comma_in_SAML_public_certificate_path():
    """`[saml] certfile` should not contain a comma (`,`).

    Because a comma is part of the API between keystone and the external
    xmlsec1 binary which utilizes the certificate, keystone cannot include a
    comma in the path to the public certificate file.
    """
    return ',' in CONF.saml.certfile