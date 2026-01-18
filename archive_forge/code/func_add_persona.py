from openchat.base import ParlaiGenerationAgent
def add_persona(self, histories, user_id, text):
    histories[user_id]['prefix'].append(f'your persona: {text}')