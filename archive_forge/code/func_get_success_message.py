from django.contrib import messages
def get_success_message(self, cleaned_data):
    return self.success_message % cleaned_data