import logging
import os
from parlai.core.agents import create_agent
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.server as server_utils
from parlai.chat_service.services.messenger.agents import MessengerAgent
from parlai.chat_service.core.socket import ChatServiceMessageSocket
from parlai.chat_service.services.messenger.message_sender import MessageSender
from parlai.chat_service.core.chat_service_manager import ChatServiceManager
def get_app_token(self):
    """
        Find and return an app access token.
        """
    if not self.opt.get('force_page_token'):
        if not os.path.exists(os.path.expanduser('~/.parlai/')):
            os.makedirs(os.path.expanduser('~/.parlai/'))
        access_token_file_path = '~/.parlai/messenger_token'
        expanded_file_path = os.path.expanduser(access_token_file_path)
        if os.path.exists(expanded_file_path):
            with open(expanded_file_path, 'r') as access_token_file:
                return access_token_file.read()
    token = input("Enter your page's access token from the developer page athttps://developers.facebook.com/apps/<YOUR APP ID>/messenger/settings/ to continue setup:")
    access_token_file_path = '~/.parlai/messenger_token'
    expanded_file_path = os.path.expanduser(access_token_file_path)
    with open(expanded_file_path, 'w+') as access_token_file:
        access_token_file.write(token)
    return token