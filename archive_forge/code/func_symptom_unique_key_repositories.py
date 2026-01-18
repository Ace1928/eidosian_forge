import keystone.conf
from keystone.common import fernet_utils as utils
from keystone.credential.providers import fernet as credential_fernet
def symptom_unique_key_repositories():
    """Key repositories for encryption should be unique.

    Even though credentials are encrypted using the same mechanism as Fernet
    tokens, they should have key repository locations that are independent of
    one another. Using the same repository to encrypt credentials and tokens
    can be considered a security vulnerability because ciphertext from the keys
    used to encrypt credentials is exposed as the token ID. Sharing a key
    repository can also lead to premature key removal during key rotation. This
    could result in indecipherable credentials, rendering them completely
    useless, or early token invalidation because the key that was used to
    encrypt the entity has been deleted.

    Ensure `keystone.conf [credential] key_repository` and `keystone.conf
    [fernet_tokens] key_repository` are not pointing to the same location.
    """
    return CONF.credential.key_repository == CONF.fernet_tokens.key_repository