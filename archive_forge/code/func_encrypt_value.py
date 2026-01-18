import configparser
import json
from typing import Any, Dict, Optional, Union
import os
import logging
from logging_config import configure_logging
from encryption_key_manager import get_valid_encryption_key
def encrypt_value(self, value: str) -> str:
    """
        Encrypts a configuration value using the configured cipher suite.

        Args:
            value (str): The value to encrypt.

        Returns:
            str: The encrypted value.
        """
    encrypted_value = self.cipher_suite.encrypt(value.encode()).decode()
    logging.debug(f'Value encrypted.')
    return encrypted_value