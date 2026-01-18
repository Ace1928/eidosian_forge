import json
import random
from parlai.tasks.blended_skill_talk.agents import raw_data_path, safe_personas_path
from parlai.tasks.interactive.worlds import InteractiveWorld as InteractiveBaseWorld
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
def finalize_episode(self):
    print('\nCHAT DONE.\n')
    if self.display_partner_persona:
        partner_persona = self.p2.replace('your persona:', "partner's persona:")
        print(f'Your partner was playing the following persona:\n{partner_persona}')
    if not self.epoch_done():
        print('\n[ Preparing new chat ... ]\n')