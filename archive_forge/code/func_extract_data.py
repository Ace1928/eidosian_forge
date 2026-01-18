import json
import sys
def extract_data(log_file):
    entries = []
    with open(log_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith('Generation Prompt:') and (not lines[i + 1].startswith('<pad>')):
                current_entry = {'prompt': '', 'response': ''}
                prompt_end_point = i + 1
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().endswith('- INFO'):
                        current_entry['prompt'] = current_entry['prompt'].rstrip()
                        break
                    current_entry['prompt'] += lines[j]
                    prompt_end_point = j
                for j in range(prompt_end_point + 1, len(lines)):
                    if lines[j].startswith('Url:') and lines[j].strip().endswith('/completions'):
                        for k in range(j + 1, len(lines)):
                            if lines[k].startswith('Data:'):
                                for l in range(k + 1, len(lines)):
                                    if 'RequestsNum: ' in lines[l]:
                                        current_entry['response'] = current_entry['response'].rstrip()
                                        entries.append(current_entry)
                                        break
                                    current_entry['response'] += lines[l]
                                else:
                                    continue
                                break
                        else:
                            continue
                        break
    return entries