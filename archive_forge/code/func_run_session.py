import random
def run_session(self):
    if self.current_role == 'interrogator':
        other_role = random.choice(['human', 'machine'])
        print(f"You are questioning a {other_role}. Try to determine if it's human or machine.")
        for _ in range(4):
            question = self.generate_question()
            print(f'Question: {question}')
            response = self.generate_response(question)
            print(f'Response: {response}')
    else:
        for _ in range(4):
            question = self.generate_question()
            response = self.generate_response(question)
            print(f'Question: {question}\nYour Response: {response}')