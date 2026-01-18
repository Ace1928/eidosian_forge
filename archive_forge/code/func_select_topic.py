import random
def select_topic(self):
    print('Select a topic to debate:')
    for index, topic in enumerate(self.topics):
        print(f'{index + 1}. {topic}')
    choice = int(input('Enter your choice (1-3): '))
    self.current_topic = self.topics[choice - 1]
    print(f'Topic selected: {self.current_topic}')