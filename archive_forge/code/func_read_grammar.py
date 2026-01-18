from pathlib import Path
def read_grammar(grammar_file_name, base_grammar_path=GRAMMAR_PATH):
    """Read grammar file from default grammar path"""
    full_path = base_grammar_path / grammar_file_name
    with open(full_path) as file:
        return file.read()