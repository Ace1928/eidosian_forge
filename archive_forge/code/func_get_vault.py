from boto.glacier.layer1 import Layer1
from boto.glacier.vault import Vault
def get_vault(self, name):
    """
        Get an object representing a named vault from Glacier. This
        operation does not check if the vault actually exists.

        :type name: str
        :param name: The name of the vault

        :rtype: :class:`boto.glacier.vault.Vault`
        :return: A Vault object representing the vault.
        """
    response_data = self.layer1.describe_vault(name)
    return Vault(self.layer1, response_data)