import json
import logging
import requests
import parlai.chat_service.utils.logging as log_utils
def delete_persona(self, persona_id):
    """
        Deletes the persona.
        """
    api_address = 'https://graph.facebook.com/' + persona_id
    response = requests.delete(api_address, params=self.auth_args)
    result = response.json()
    log_utils.print_and_log(logging.INFO, '"Facebook response from delete persona: {}"'.format(result))
    return result