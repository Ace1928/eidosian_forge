import collections
import types
import numpy as np
from ..utils import (
from .base import ArgumentHandler, Dataset, Pipeline, PipelineException, build_pipeline_init_args
def sequential_inference(self, **inputs):
    """
        Inference used for models that need to process sequences in a sequential fashion, like the SQA models which
        handle conversational query related to a table.
        """
    if self.framework == 'pt':
        all_logits = []
        all_aggregations = []
        prev_answers = None
        batch_size = inputs['input_ids'].shape[0]
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        token_type_ids = inputs['token_type_ids'].to(self.device)
        token_type_ids_example = None
        for index in range(batch_size):
            if prev_answers is not None:
                prev_labels_example = token_type_ids_example[:, 3]
                model_labels = np.zeros_like(prev_labels_example.cpu().numpy())
                token_type_ids_example = token_type_ids[index]
                for i in range(model_labels.shape[0]):
                    segment_id = token_type_ids_example[:, 0].tolist()[i]
                    col_id = token_type_ids_example[:, 1].tolist()[i] - 1
                    row_id = token_type_ids_example[:, 2].tolist()[i] - 1
                    if row_id >= 0 and col_id >= 0 and (segment_id == 1):
                        model_labels[i] = int(prev_answers[col_id, row_id])
                token_type_ids_example[:, 3] = torch.from_numpy(model_labels).type(torch.long).to(self.device)
            input_ids_example = input_ids[index]
            attention_mask_example = attention_mask[index]
            token_type_ids_example = token_type_ids[index]
            outputs = self.model(input_ids=input_ids_example.unsqueeze(0), attention_mask=attention_mask_example.unsqueeze(0), token_type_ids=token_type_ids_example.unsqueeze(0))
            logits = outputs.logits
            if self.aggregate:
                all_aggregations.append(outputs.logits_aggregation)
            all_logits.append(logits)
            dist_per_token = torch.distributions.Bernoulli(logits=logits)
            probabilities = dist_per_token.probs * attention_mask_example.type(torch.float32).to(dist_per_token.probs.device)
            coords_to_probs = collections.defaultdict(list)
            for i, p in enumerate(probabilities.squeeze().tolist()):
                segment_id = token_type_ids_example[:, 0].tolist()[i]
                col = token_type_ids_example[:, 1].tolist()[i] - 1
                row = token_type_ids_example[:, 2].tolist()[i] - 1
                if col >= 0 and row >= 0 and (segment_id == 1):
                    coords_to_probs[col, row].append(p)
            prev_answers = {key: np.array(coords_to_probs[key]).mean() > 0.5 for key in coords_to_probs}
        logits_batch = torch.cat(tuple(all_logits), 0)
        return (logits_batch,) if not self.aggregate else (logits_batch, torch.cat(tuple(all_aggregations), 0))
    else:
        all_logits = []
        all_aggregations = []
        prev_answers = None
        batch_size = inputs['input_ids'].shape[0]
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids'].numpy()
        token_type_ids_example = None
        for index in range(batch_size):
            if prev_answers is not None:
                prev_labels_example = token_type_ids_example[:, 3]
                model_labels = np.zeros_like(prev_labels_example, dtype=np.int32)
                token_type_ids_example = token_type_ids[index]
                for i in range(model_labels.shape[0]):
                    segment_id = token_type_ids_example[:, 0].tolist()[i]
                    col_id = token_type_ids_example[:, 1].tolist()[i] - 1
                    row_id = token_type_ids_example[:, 2].tolist()[i] - 1
                    if row_id >= 0 and col_id >= 0 and (segment_id == 1):
                        model_labels[i] = int(prev_answers[col_id, row_id])
                token_type_ids_example[:, 3] = model_labels
            input_ids_example = input_ids[index]
            attention_mask_example = attention_mask[index]
            token_type_ids_example = token_type_ids[index]
            outputs = self.model(input_ids=np.expand_dims(input_ids_example, axis=0), attention_mask=np.expand_dims(attention_mask_example, axis=0), token_type_ids=np.expand_dims(token_type_ids_example, axis=0))
            logits = outputs.logits
            if self.aggregate:
                all_aggregations.append(outputs.logits_aggregation)
            all_logits.append(logits)
            dist_per_token = tfp.distributions.Bernoulli(logits=logits)
            probabilities = dist_per_token.probs_parameter() * tf.cast(attention_mask_example, tf.float32)
            coords_to_probs = collections.defaultdict(list)
            token_type_ids_example = token_type_ids_example
            for i, p in enumerate(tf.squeeze(probabilities).numpy().tolist()):
                segment_id = token_type_ids_example[:, 0].tolist()[i]
                col = token_type_ids_example[:, 1].tolist()[i] - 1
                row = token_type_ids_example[:, 2].tolist()[i] - 1
                if col >= 0 and row >= 0 and (segment_id == 1):
                    coords_to_probs[col, row].append(p)
            prev_answers = {key: np.array(coords_to_probs[key]).mean() > 0.5 for key in coords_to_probs}
        logits_batch = tf.concat(tuple(all_logits), 0)
        return (logits_batch,) if not self.aggregate else (logits_batch, tf.concat(tuple(all_aggregations), 0))