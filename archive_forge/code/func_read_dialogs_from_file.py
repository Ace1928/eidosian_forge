import json
def read_dialogs_from_file(file_path):
    with open(file_path, 'r') as file:
        dialogs = json.load(file)
    return dialogs