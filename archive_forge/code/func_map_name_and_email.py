from email.utils import parseaddr
def map_name_and_email(self, name, email):
    """Map a name and an email to the preferred name and email.

        :param name: the current name
        :param email: the current email
        :result: the preferred name and email
        """
    try:
        new_name, new_email = self._user_map[name, email]
    except KeyError:
        new_name = name
        if self._default_domain and (not email):
            new_email = b'%s@%s' % (name, self._default_domain)
        else:
            new_email = email
    return (new_name, new_email)