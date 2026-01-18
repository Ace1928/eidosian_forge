from __future__ import absolute_import, division, print_function
import os
def get_vault_client(self, hashi_vault_logout_inferred_token=True, hashi_vault_revoke_on_logout=False, **kwargs):
    """
        creates a Vault client with the given kwargs

        :param hashi_vault_logout_inferred_token: if True performs "logout" after creation to remove any token that
        the hvac library itself may have read in. Only used if "token" is not included in kwargs.
        :type hashi_vault_logout_implied_token: bool

        :param hashi_vault_revoke_on_logout: if True revokes any current token on logout. Only used if a logout is performed. Not recommended.
        :type hashi_vault_revoke_on_logout: bool
        """
    client = hvac.Client(**kwargs)
    if hashi_vault_logout_inferred_token and 'token' not in kwargs:
        client.logout(revoke_token=hashi_vault_revoke_on_logout)
    return client