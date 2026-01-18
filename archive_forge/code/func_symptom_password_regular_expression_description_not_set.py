import re
import keystone.conf
def symptom_password_regular_expression_description_not_set():
    """Password regular expression description is not set.

    The password regular expression is set, but the description is not. Thus,
    if a user fails the password regular expression, they will not receive a
    message to explain why their requested password was insufficient.

    Ensure `[security_compliance] password_regex_description` is set with a
    description of your password regular expression in a language for humans.
    """
    return CONF.security_compliance.password_regex and (not CONF.security_compliance.password_regex_description)