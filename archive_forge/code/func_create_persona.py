import json
import logging
import requests
import parlai.chat_service.utils.logging as log_utils
def create_persona(self, name, image_url):
    """
        Creates a new persona and returns persona_id.
        """
    api_address = 'https://graph.facebook.com/me/personas'
    message = {'name': name, 'profile_picture_url': image_url}
    response = requests.post(api_address, params=self.auth_args, json=message)
    result = response.json()
    log_utils.print_and_log(logging.INFO, '"Facebook response from create persona: {}"'.format(result))
    return result