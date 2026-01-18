import logging
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import time_util
from saml2.attribute_converter import to_local
from saml2.response import IncorrectlySigned
from saml2.s_utils import OtherError
from saml2.s_utils import VersionMismatch
from saml2.sigver import verify_redirect_signature
from saml2.validate import NotValid
from saml2.validate import valid_instance
def subject_id(self):
    """The name of the subject can be in either of
        BaseID, NameID or EncryptedID

        :return: The identifier if there is one
        """
    if 'subject' in self.message.keys():
        _subj = self.message.subject
        if 'base_id' in _subj.keys() and _subj.base_id:
            return _subj.base_id
        elif _subj.name_id:
            return _subj.name_id
    elif 'base_id' in self.message.keys() and self.message.base_id:
        return self.message.base_id
    elif self.message.name_id:
        return self.message.name_id
    else:
        pass