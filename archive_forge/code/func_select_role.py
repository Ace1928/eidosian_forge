import random
def select_role(self):
    print('Select a role to play:')
    for index, role in enumerate(self.roles):
        print(f'{index + 1}. {role}')
    choice = int(input('Enter your choice (1-3): '))
    self.current_role = self.roles[choice - 1]
    print(f'You are now the {self.current_role}.')